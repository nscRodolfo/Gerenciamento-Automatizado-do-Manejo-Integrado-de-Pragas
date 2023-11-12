package my.aplication.manejointeligentedepragas;

import android.content.Intent;
import android.content.SharedPreferences;

//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;

//import android.support.design.widget.NavigationView;
import com.google.android.material.navigation.NavigationView;

//import android.support.v4.view.GravityCompat;
import androidx.core.view.GravityCompat;

//import android.support.v4.widget.DrawerLayout;
import androidx.drawerlayout.widget.DrawerLayout;

//import android.support.v7.app.ActionBarDrawerToggle;
import androidx.appcompat.app.ActionBarDrawerToggle;

//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;

//import android.support.v7.widget.Toolbar;
import androidx.appcompat.widget.Toolbar;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PlanoAmostragem;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PresencaPraga;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;
import my.aplication.manejointeligentedepragas.model.PlanoAmostragemModel;
import my.aplication.manejointeligentedepragas.model.PresencaPragaModel;

import com.example.manejointeligentedepragas.R;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

import com.zplesac.connectionbuddy.ConnectionBuddy;
import com.zplesac.connectionbuddy.ConnectionBuddyConfiguration;
import com.zplesac.connectionbuddy.interfaces.ConnectivityChangeListener;
import com.zplesac.connectionbuddy.models.ConnectivityEvent;

public class AdicionarCultura extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{


    //String nomePlantas[];
    //String tamanhoTalhao[];

    private DrawerLayout drawerLayout;

    ArrayList<String> nomePlantas = new ArrayList<String>();
    ArrayList<String> tamanhoTalhao = new ArrayList<String>();
    ArrayList<Integer> Cod_Planta = new ArrayList<Integer>();
    Float tamanhoSelecionado;
    Integer codigoSelecionado;

    Integer numeroDeTalhoes;
    
    EditText ETtamanhoCulturaUsuario;

    TextView TVnumeroTalhoesSugerido;

    Integer Cod_Propriedade;

    Button salvarCultura;

    Boolean click = true; // verificação de click (nao adicionar dados duplos)

    ArrayList<String> plantasadd = new ArrayList<String>(); // para remover cultura ja adicionada

    String nomeSelecionado;

    String nomePropriedade;

    ArrayList<PresencaPragaModel> presencaPragaModels = new ArrayList();

    ArrayList<PlanoAmostragemModel> planoAmostragemModels = new ArrayList();



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_adicionar_cultura);


        ConnectionBuddyConfiguration networkInspectorConfiguration = new ConnectionBuddyConfiguration.Builder(this).build();
        ConnectionBuddy.getInstance().init(networkInspectorConfiguration);

        nomePropriedade = getIntent().getStringExtra("nomePropriedade");
        Cod_Propriedade = getIntent().getIntExtra("Cod_Propriedade", 0);
        plantasadd = getIntent().getStringArrayListExtra("plantasadd");

        ETtamanhoCulturaUsuario = findViewById(R.id.etPpT);
        TVnumeroTalhoesSugerido = findViewById(R.id.tvNumeroTalhoes);

        salvarCultura = findViewById(R.id.btnSalvrCultura);

        Spinner dropdown = findViewById(R.id.dropdownCultura);
        ResgatarPlantas(dropdown);

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_add_cultura);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_add_cultura);
        NavigationView navigationView = findViewById(R.id.nav_view_add_cultura);
        navigationView.setNavigationItemSelectedListener(this);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();
        View headerView = navigationView.getHeaderView(0);

        Controller_Usuario controller_usuario = new Controller_Usuario(getBaseContext());
        String nomeUsu = controller_usuario.getUser().getNome();
        String emailUsu = controller_usuario.getUser().getEmail();

        TextView nomeMenu = headerView.findViewById(R.id.nomeMenu);
        nomeMenu.setText(nomeUsu);

        TextView emailMenu = headerView.findViewById(R.id.emailMenu);
        emailMenu.setText(emailUsu);

        setTitle("MIP² | "+nomePropriedade);

        salvarCultura.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(click) {
                    SalvarCultura();
                }else{
                    Toast.makeText(AdicionarCultura.this, "Aguarde um momento..." ,Toast.LENGTH_LONG).show();
                }
            }
        });

        dropdown.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                tamanhoSelecionado = Float.parseFloat(tamanhoTalhao.get(position));
                codigoSelecionado = Cod_Planta.get(position);
                nomeSelecionado = nomePlantas.get(position);
                //teste Toast.makeText(parent.getContext(), "Selected: " + tamanhoSelecionado + " Código Planta:" + codigoSelecionado,Toast.LENGTH_LONG).show();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }
        });

        ETtamanhoCulturaUsuario.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {

            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {

            }

            @Override
            public void afterTextChanged(Editable s) {
                if(s.length()!=0){
                    Float tamanhoDigitado = Float.parseFloat(s.toString());
                    if(tamanhoDigitado <= tamanhoSelecionado) {
                        TVnumeroTalhoesSugerido.setText("1");
                        numeroDeTalhoes = 1;
                    }else{
                        Float numeroSugerido = tamanhoDigitado / tamanhoSelecionado;
                        Double a = Math.ceil(numeroSugerido) ;
                        TVnumeroTalhoesSugerido.setText("" + a.intValue());

                        numeroDeTalhoes = a.intValue();
                    }
                }else{
                    TVnumeroTalhoesSugerido.setText("0");
                    numeroDeTalhoes = 0;
                }


            }
        });
    }


    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        ConnectionBuddy.getInstance().registerForConnectivityEvents(this, new ConnectivityChangeListener() {
            @Override
            public void onConnectionChange(ConnectivityEvent event) {
                Utils u = new Utils();
                if(!u.isConected(getBaseContext()))
                {
                    //Toast.makeText(AcoesCultura.this,"Você está offline!", Toast.LENGTH_LONG).show();
                }else{
                    final Controller_PlanoAmostragem cpa = new Controller_PlanoAmostragem(AdicionarCultura.this);
                    final Controller_PresencaPraga cpp = new Controller_PresencaPraga(AdicionarCultura.this);

                    //Toast.makeText(AcoesCultura.this,"Você está online!", Toast.LENGTH_LONG).show();

                    planoAmostragemModels = cpa.getPlanoOffline();
                    presencaPragaModels = cpp.getPresencaPragaOffline();

                    for(int i=0; i<planoAmostragemModels.size(); i++){
                        SalvarPlanos(planoAmostragemModels.get(i));
                    }
                    cpa.removerPlano();

                    for(int i=0; i<presencaPragaModels.size(); i++){
                        SalvarPresencas(presencaPragaModels.get(i));
                    }
                    cpp.updatePresencaSyncStatus();


                }
            }
        });
    }

    @Override
    protected void onStop() {
        super.onStop();
        ConnectionBuddy.getInstance().unregisterFromConnectivityEvents(this);
    }


    public void SalvarPlanos(PlanoAmostragemModel pam){
        Controller_Usuario cu = new Controller_Usuario(AdicionarCultura.this);
        String Autor = cu.getUser().getNome();

        String url = "https://mip.software/phpapp/salvaPlanoAmostragem.php?Cod_Talhao=" + pam.getFk_Cod_Talhao()
                +"&&Data="+pam.getDate()
                +"&&PlantasInfestadas="+pam.getPlantasInfestadas()
                +"&&PlantasAmostradas="+pam.getPlantasAmostradas()
                +"&&Cod_Praga="+pam.getFk_Cod_Praga()
                +"&&Autor="+Autor;

        RequestQueue queue = Volley.newRequestQueue(AdicionarCultura.this);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {


            }

        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(AdicionarCultura.this,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }

    public void SalvarPresencas(PresencaPragaModel ppm){
        String url = "https://mip.software/phpapp/updatePraga.php?Cod_Praga="+ppm.getFk_Cod_Praga()+
                "&&Cod_Talhao="+ppm.getFk_Cod_Talhao()+"&&Status="+ppm.getStatus();
        RequestQueue queue = Volley.newRequestQueue(AdicionarCultura.this);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {


            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(AdicionarCultura.this,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
        switch (menuItem.getItemId()){
            case R.id.drawerPerfil:
                Intent i= new Intent(this, Perfil.class);
                startActivity(i);
                break;
            case R.id.drawerProp:
                Intent prop= new Intent(this, Propriedades.class);
                startActivity(prop);
                break;

            case R.id.drawerPlantas:
                Intent j = new Intent(this, VisualizaPlantas.class);
                startActivity(j);
                break;

            case R.id.drawerPrag:
                Intent k = new Intent(this, VisualizaPragas.class);
                startActivity(k);
                break;

            case R.id.drawerMet:
                Intent l = new Intent(this, VisualizaMetodos.class);
                startActivity(l);
                break;

            case R.id.drawerSobreMip:
                Intent p = new Intent(this, SobreMIP.class);
                startActivity(p);
                break;

            case R.id.drawerTutorial:
                SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
                SharedPreferences.Editor editor = pref.edit();
                editor.putBoolean("isIntroOpened",false);
                editor.commit();

                Intent intro = new Intent(this, IntroActivity.class);
                startActivity(intro);
                break;

            case R.id.drawerSobre:
                Intent pp = new Intent(this, Sobre.class);
                startActivity(pp);
                break;
            case R.id.drawerReferencias:
                Intent pi = new Intent(this, Referencias.class);
                startActivity(pi);
                break;

            case R.id.drawerRecomendações:
                Intent pa = new Intent(this, RecomendacoesMAPA.class);
                startActivity(pa);
                break;
        }
        drawerLayout.closeDrawer(GravityCompat.START);
        return true;
    }

    public void ResgatarPlantas(final Spinner dropdown){
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            Toast.makeText(this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
        }else { // se tem acesso à internet
            String url = "https://mip.software/phpapp/selecionarCulturaTamanho.php";


            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj = array.getJSONObject(i);

                            tamanhoTalhao.add(obj.getString("TamanhoTalhao"));
                            nomePlantas.add(obj.getString("Nome"));
                            Cod_Planta.add(obj.getInt("Cod_Planta"));


                        }
                       // nomePlantas.removeAll(plantasadd);

                        //String[] items = new String[nomePlantas.size()];
                        //items = nomePlantas.toArray(items);
                        ArrayAdapter<String> adapter = new ArrayAdapter<String>(getBaseContext(), android.R.layout.simple_spinner_dropdown_item, nomePlantas);
                        dropdown.setAdapter(adapter);


                    } catch (JSONException e) {
                        Toast.makeText(AdicionarCultura.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(AdicionarCultura.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));

        }
    }

    public void SalvarCultura(){
        final String tamanhoCultura = ETtamanhoCulturaUsuario.getText().toString();


        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            Toast.makeText(this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();

        }else if(tamanhoCultura.isEmpty()) {
            Toast.makeText(this,"Informe o tamanho de sua cultura (em hectare).", Toast.LENGTH_LONG).show();

        }else if(plantasadd.contains(nomeSelecionado)){
            Toast.makeText(this,"Esta cultura já existe em sua propriedade.", Toast.LENGTH_LONG).show();

        }else{ // se tem acesso à internet
            click = false;
            String url = "https://mip.software/phpapp/adicionarCultura.php?TamanhoDaCultura="+tamanhoCultura+
                    "&&fk_Propriedade_Cod_Propriedade="+Cod_Propriedade+
                    "&&fk_Planta_Cod_Planta="+codigoSelecionado+
                    "&&qtdTalhao="+numeroDeTalhoes;
            //Toast.makeText(this,url, Toast.LENGTH_LONG).show();

            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();

                        JSONObject obj1 = new JSONObject(response);
                        boolean confirmacao = obj1.getBoolean("confirmacao");

                        if(confirmacao){
                            Intent k = new Intent(AdicionarCultura.this, Cultura.class);
                            k.putExtra("Cod_Propriedade", Cod_Propriedade);
                            k.putExtra("nomePropriedade", nomePropriedade);
                            startActivity(k);
                        }else{
                            Toast.makeText(AdicionarCultura.this, "Cultura não cadastrada! Tente novamente",Toast.LENGTH_LONG).show();
                        }
                    } catch (JSONException e) {
                        Toast.makeText(AdicionarCultura.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(AdicionarCultura.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));

        }
    }

}
