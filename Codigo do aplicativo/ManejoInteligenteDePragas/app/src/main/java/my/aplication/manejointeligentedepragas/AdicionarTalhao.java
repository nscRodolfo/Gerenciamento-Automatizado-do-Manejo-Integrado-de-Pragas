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
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
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

import org.json.JSONException;
import org.json.JSONObject;

import com.zplesac.connectionbuddy.ConnectionBuddy;
import com.zplesac.connectionbuddy.ConnectionBuddyConfiguration;
import com.zplesac.connectionbuddy.interfaces.ConnectivityChangeListener;
import com.zplesac.connectionbuddy.models.ConnectivityEvent;

import java.util.ArrayList;


public class AdicionarTalhao extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{

    private DrawerLayout drawerLayout;

    EditText nomeT;
    Button salvarTalhao;

    Integer Cod_Propriedade;
    String nomePropriedade;
    boolean aplicado;
    int codCultura;
    String nome;
    int CodT;
    String NomeTalhao;

    Boolean editar = false;
    Boolean click = true; // verificação de click (nao adicionar dados duplos)


    ArrayList<PresencaPragaModel> presencaPragaModels = new ArrayList();

    ArrayList<PlanoAmostragemModel> planoAmostragemModels = new ArrayList();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_adicionar_talhao);

        ConnectionBuddyConfiguration networkInspectorConfiguration = new ConnectionBuddyConfiguration.Builder(this).build();
        ConnectionBuddy.getInstance().init(networkInspectorConfiguration);

        Cod_Propriedade = getIntent().getIntExtra("Cod_Propriedade", 0);
        nomePropriedade = getIntent().getStringExtra("nomePropriedade");
        codCultura = getIntent().getIntExtra("Cod_Cultura", 0);
        nome = getIntent().getStringExtra("NomeCultura");
        aplicado = getIntent().getBooleanExtra("Aplicado", false);
        CodT = getIntent().getIntExtra("Cod_Talhao", 0);
        NomeTalhao = getIntent().getStringExtra("NomeTalhao");

        nomeT = findViewById(R.id.etNomeTalhoes);
        salvarTalhao = findViewById(R.id.btnSalvarTalhao);

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_add_talhao);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_add_tallhao);
        NavigationView navigationView = findViewById(R.id.nav_view_add_talhao);
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

        if(CodT != 0){
            editar = true;
            nomeT.setText(NomeTalhao);
        }

        salvarTalhao.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if(click) {
                    if(editar){
                        EditarTalhao();
                    }else{
                        SalvarTalhao();
                    }
                }else{
                    Toast.makeText(AdicionarTalhao.this, "Aguarde um momento..." ,Toast.LENGTH_LONG).show();
                }
            }
        });



    }

    public void EditarTalhao(){
        final String nomeTalhao = nomeT.getText().toString();
        if(nome.isEmpty()){
            Toast.makeText(AdicionarTalhao.this, "Nome do talhão é obrigatório!" ,Toast.LENGTH_LONG).show();
        }else{
            String url = "https://mip.software/phpapp/editarTalhao.php?Cod_Talhao=" + CodT +
                    "&&NomeTalhao="+nomeTalhao ;

            RequestQueue queue = Volley.newRequestQueue(AdicionarTalhao.this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        JSONObject obj = new JSONObject(response);
                        boolean confirmacao = obj.getBoolean("confirmacao");
                        if(confirmacao){
                            Intent k = new Intent(AdicionarTalhao.this, Talhoes.class);
                            k.putExtra("Cod_Propriedade", Cod_Propriedade);
                            k.putExtra("nomePropriedade", nomePropriedade);
                            k.putExtra("Cod_Cultura", codCultura);
                            k.putExtra("NomeCultura", nome);
                            k.putExtra("Aplicado", aplicado);
                            Toast.makeText(AdicionarTalhao.this, "Talhao alterado com sucesso!",Toast.LENGTH_LONG).show();
                            startActivity(k);
                        }else{
                            Toast.makeText(AdicionarTalhao.this, "Talhão não editado! Tente novamente",Toast.LENGTH_LONG).show();
                        }

                    } catch (JSONException e) {
                        Toast.makeText(AdicionarTalhao.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(AdicionarTalhao.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));
        }
    }

    public void SalvarTalhao(){
        final String nomeTalhao = nomeT.getText().toString();
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            Toast.makeText(this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
        }else if(nomeTalhao.isEmpty()){
            Toast.makeText(this,"Seu Talhão deve ter um nome", Toast.LENGTH_LONG).show();

        }else{ // se tem acesso à internet
            click = false;
            String url = "https://mip.software/phpapp/adicionarTalhao.php?NomeTalhao="+nomeTalhao+"&&Cod_Cultura="+codCultura;
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
                            Intent k = new Intent(AdicionarTalhao.this, Talhoes.class);
                            k.putExtra("Cod_Propriedade", Cod_Propriedade);
                            k.putExtra("nomePropriedade", nomePropriedade);
                            k.putExtra("Cod_Cultura", codCultura);
                            k.putExtra("NomeCultura", nome);
                            k.putExtra("Aplicado", aplicado);
                            startActivity(k);
                        }else{
                            Toast.makeText(AdicionarTalhao.this, "Talhão não cadastrado! Tente novamente",Toast.LENGTH_LONG).show();
                        }
                    } catch (JSONException e) {
                        Toast.makeText(AdicionarTalhao.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(AdicionarTalhao.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));

        }


    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            Intent k = new Intent(AdicionarTalhao.this, Talhoes.class);
            k.putExtra("Cod_Propriedade", Cod_Propriedade);
            k.putExtra("nomePropriedade", nomePropriedade);
            k.putExtra("Cod_Cultura", codCultura);
            k.putExtra("NomeCultura", nome);
            k.putExtra("Aplicado", aplicado);
            this.startActivity(k);
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
                    final Controller_PlanoAmostragem cpa = new Controller_PlanoAmostragem(AdicionarTalhao.this);
                    final Controller_PresencaPraga cpp = new Controller_PresencaPraga(AdicionarTalhao.this);

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
        Controller_Usuario cu = new Controller_Usuario(AdicionarTalhao.this);
        String Autor = cu.getUser().getNome();

        String url = "https://mip.software/phpapp/salvaPlanoAmostragem.php?Cod_Talhao=" + pam.getFk_Cod_Talhao()
                +"&&Data="+pam.getDate()
                +"&&PlantasInfestadas="+pam.getPlantasInfestadas()
                +"&&PlantasAmostradas="+pam.getPlantasAmostradas()
                +"&&Cod_Praga="+pam.getFk_Cod_Praga()
                +"&&Autor="+Autor;

        RequestQueue queue = Volley.newRequestQueue(AdicionarTalhao.this);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {


            }

        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(AdicionarTalhao.this,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }

    public void SalvarPresencas(PresencaPragaModel ppm){
        String url = "https://mip.software/phpapp/updatePraga.php?Cod_Praga="+ppm.getFk_Cod_Praga()+
                "&&Cod_Talhao="+ppm.getFk_Cod_Talhao()+"&&Status="+ppm.getStatus();
        RequestQueue queue = Volley.newRequestQueue(AdicionarTalhao.this);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {


            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(AdicionarTalhao.this,error.toString(), Toast.LENGTH_LONG).show();
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

}
