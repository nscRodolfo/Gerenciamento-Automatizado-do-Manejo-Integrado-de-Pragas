package my.aplication.manejointeligentedepragas;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.drawable.ColorDrawable;


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
//import android.support.design.widget.FloatingActionButton;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
//import android.support.v7.widget.LinearLayoutManager;
import androidx.recyclerview.widget.LinearLayoutManager;
//import android.support.v7.widget.RecyclerView;
import androidx.recyclerview.widget.RecyclerView;

import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Atinge;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PlanoAmostragem;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Praga;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_PresencaPraga;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Talhao;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;

import com.example.manejointeligentedepragas.R;

import my.aplication.manejointeligentedepragas.RecyclerViewAdapter.TalhaoCardAdapter;
import my.aplication.manejointeligentedepragas.model.AtingeModel;
import my.aplication.manejointeligentedepragas.model.PlanoAmostragemModel;
import my.aplication.manejointeligentedepragas.model.PragaModel;
import my.aplication.manejointeligentedepragas.model.PresencaPragaModel;
import my.aplication.manejointeligentedepragas.model.TalhaoModel;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

import com.zplesac.connectionbuddy.ConnectionBuddy;
import com.zplesac.connectionbuddy.ConnectionBuddyConfiguration;
import com.zplesac.connectionbuddy.interfaces.ConnectivityChangeListener;
import com.zplesac.connectionbuddy.models.ConnectivityEvent;

public class Talhoes extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{

    public FloatingActionButton fabAddTalhao;
    public TextView tvAddTalhao;
    public String tipoUsu;
    double nivelControle;

    private static final String TAG = "Talhao";
    private ArrayList<TalhaoModel> cards = new ArrayList<>();
    private ArrayList<TalhaoModel> web = new ArrayList<>();
    private ArrayList<TalhaoModel> local = new ArrayList<>();

    private ArrayList<PragaModel> webPraga = new ArrayList<>();
    private ArrayList<PragaModel> localPraga = new ArrayList<>();

    Integer Cod_Propriedade;
    String nomePropriedade;
    Integer Cod_Planta;

    private Dialog mDialog;

    private DrawerLayout drawerLayout;

    boolean aplicado;
    int codCultura;
    String nome;

    ArrayList<PresencaPragaModel> presencaPragaModels = new ArrayList();

    ArrayList<PlanoAmostragemModel> planoAmostragemModels = new ArrayList();


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_talhoes);

        ConnectionBuddyConfiguration networkInspectorConfiguration = new ConnectionBuddyConfiguration.Builder(this).build();
        ConnectionBuddy.getInstance().init(networkInspectorConfiguration);

        Cod_Propriedade = getIntent().getIntExtra("Cod_Propriedade", 0);
        nomePropriedade = getIntent().getStringExtra("nomePropriedade");
        codCultura = getIntent().getIntExtra("Cod_Cultura", 0);
        nome = getIntent().getStringExtra("NomeCultura");
        Cod_Planta = getIntent().getIntExtra("Cod_Planta",0);

        openDialog();

        tvAddTalhao = findViewById(R.id.tvAddTalhao);
        fabAddTalhao = findViewById(R.id.fabAddTalhao);

        regataPragasDaCultura();

        Controller_Usuario cu = new Controller_Usuario(getBaseContext());
        tipoUsu = cu.getUser().getTipo();

        Utils u = new Utils();
        if(!u.isConected(getBaseContext())) {
            Controller_Talhao ct = new Controller_Talhao(Talhoes.this);
            cards = ct.getTalhao(codCultura);
            if(cards.size() == 0){
                AlertDialog.Builder dlgBox = new AlertDialog.Builder(Talhoes.this);
                dlgBox.setTitle("Aviso");
                dlgBox.setMessage("Para possuir os dados desses talhões enquanto está sem internet, você precisa acessar essa página online pelo menos uma vez.");
                dlgBox.setPositiveButton("Entendi", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {

                    }
                });
                dlgBox.show();
            }
            tvAddTalhao.setVisibility(View.GONE);
            fabAddTalhao.hide();
            mDialog.dismiss();
            iniciarRecyclerView();
        }else{
            if(tipoUsu.equals("Funcionario")){
                tvAddTalhao.setVisibility(View.GONE);
                fabAddTalhao.hide();
                resgatarDados();
            }else if(tipoUsu.equals("Produtor")){
                resgatarDados();
            }
        }


        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_talhoes);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_talhoes);
        NavigationView navigationView = findViewById(R.id.nav_view_talhoes);
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

        fabAddTalhao.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Utils u = new Utils();
                if(!u.isConected(getBaseContext()))
                {
                    Toast.makeText(Talhoes.this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
                    mDialog.dismiss();
                }else {
                    Intent k = new Intent(Talhoes.this, AdicionarTalhao.class);
                    k.putExtra("Cod_Propriedade", Cod_Propriedade);
                    k.putExtra("nomePropriedade", nomePropriedade);
                    k.putExtra("Cod_Cultura", codCultura);
                    k.putExtra("NomeCultura", nome);
                    k.putExtra("Aplicado", aplicado);
                    startActivity(k);
                }
            }
        });

        tvAddTalhao.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Utils u = new Utils();
                if(!u.isConected(getBaseContext()))
                {
                    Toast.makeText(Talhoes.this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
                    mDialog.dismiss();
                }else {
                    Intent k = new Intent(Talhoes.this, AdicionarTalhao.class);
                    k.putExtra("Cod_Propriedade", Cod_Propriedade);
                    k.putExtra("nomePropriedade", nomePropriedade);
                    k.putExtra("Cod_Cultura", codCultura);
                    k.putExtra("NomeCultura", nome);
                    k.putExtra("Aplicado", aplicado);
                    startActivity(k);
                }
            }
        });
    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            Intent i = new Intent(Talhoes.this,Cultura.class);
            i.putExtra("Cod_Propriedade", Cod_Propriedade);
            i.putExtra("nomePropriedade", nomePropriedade);
            startActivity(i);
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
                    final Controller_PlanoAmostragem cpa = new Controller_PlanoAmostragem(Talhoes.this);
                    final Controller_PresencaPraga cpp = new Controller_PresencaPraga(Talhoes.this);

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
        Controller_Usuario cu = new Controller_Usuario(Talhoes.this);
        String Autor = cu.getUser().getNome();

        String url = "https://mip.software/phpapp/salvaPlanoAmostragem.php?Cod_Talhao=" + pam.getFk_Cod_Talhao()
                +"&&Data="+pam.getDate()
                +"&&PlantasInfestadas="+pam.getPlantasInfestadas()
                +"&&PlantasAmostradas="+pam.getPlantasAmostradas()
                +"&&Cod_Praga="+pam.getFk_Cod_Praga()
                +"&&Autor="+Autor;

        RequestQueue queue = Volley.newRequestQueue(Talhoes.this);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {


            }

        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(Talhoes.this,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }

    public void SalvarPresencas(PresencaPragaModel ppm){
        String url = "https://mip.software/phpapp/updatePraga.php?Cod_Praga="+ppm.getFk_Cod_Praga()+
                "&&Cod_Talhao="+ppm.getFk_Cod_Talhao()+"&&Status="+ppm.getStatus();
        RequestQueue queue = Volley.newRequestQueue(Talhoes.this);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {


            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(Talhoes.this,error.toString(), Toast.LENGTH_LONG).show();
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

    private void resgatarDados(){
        //Log.d(TAG, "resgatarDados: resgatou");

        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            Toast.makeText(this,"Habilite a conexão com a internet", Toast.LENGTH_LONG).show();
            mDialog.dismiss();
        }else { // se tem acesso à internet

            String url = "https://mip.software/phpapp/resgatarTalhoesTela.php?Cod_Cultura=" + codCultura;

            RequestQueue queue = Volley.newRequestQueue(Talhoes.this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj = array.getJSONObject(i);
                            TalhaoModel u = new TalhaoModel();
                            u.setCod_Talhao(obj.getInt("Cod_Talhao"));
                            u.setFk_Cod_Cultura(obj.getInt("fk_Cultura_Cod_Cultura"));
                            u.setFk_Cod_Planta(obj.getInt("fk_Planta_Cod_Planta"));
                            u.setNome(obj.getString("Nome"));
                            //resgataDataFimContagem();//resgatar a data fim passando pro php(dias pra contagem) o cod do talhao
                            //setar no objeto
                            //no talhao card adapter fazer a conta da diferença e tratar as esxeções iguais ao php
                            boolean auxAplicado;
                            if(obj.getInt("Aplicado") != 0){
                                auxAplicado = true;
                            }else{
                                auxAplicado = false;
                            }
                            u.setAplicado(auxAplicado);
                            web.add(u);
                        }

                        Controller_Talhao ct= new Controller_Talhao(Talhoes.this);
                        local = ct.getTalhao(codCultura);

                        //Toast.makeText(Talhoes.this, "AQui", Toast.LENGTH_LONG).show();
                        if(web.size() == local.size()) {
                            if(!web.equals(local)) {
                                local = web;
                                for (int i = 0; i < local.size(); i++) {
                                    ct.removerTalhaoEspecifica(local.get(i));
                                    ct.addTalhao(local.get(i));
                                }
                                cards = local;
                            }else{
                                cards = local;
                            }
                        }else{
                            local = web;
                            for (int i = 0; i < local.size(); i++) {
                                ct.removerTalhaoEspecifica(local.get(i));
                                ct.addTalhao(local.get(i));
                            }
                            cards = local;
                        }
                        iniciarRecyclerView();
                        mDialog.dismiss();
                    } catch (JSONException e) {
                        Toast.makeText(Talhoes.this, e.toString(), Toast.LENGTH_LONG).show();
                        mDialog.dismiss();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(Talhoes.this,error.toString(), Toast.LENGTH_LONG).show();
                    mDialog.dismiss();
                }
            }));

        }
    }

    private void regataPragasDaCultura(){
        final Controller_Praga cp = new Controller_Praga(Talhoes.this);
        final Controller_Atinge ca = new Controller_Atinge(Talhoes.this);
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            localPraga = cp.getPragaAtinge(Cod_Planta);
            mDialog.dismiss();
        }else { // se tem acesso à internet

            String url = "https://mip.software/phpapp/ResgataPrafasDaCultura.php?Cod_Cultura=" + codCultura;

            RequestQueue queue = Volley.newRequestQueue(Talhoes.this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj = array.getJSONObject(i);
                            PragaModel u = new PragaModel();
                            AtingeModel a = new AtingeModel();
                            u.setCod_Praga(obj.getInt("Cod_Praga"));
                            u.setNome(obj.getString("Nome"));
                            u.setFamilia(obj.getString("Familia"));
                            u.setOrdem(obj.getString("Ordem"));
                            u.setDescricao(obj.getString("Descricao"));
                            u.setNomeCientifico(obj.getString("NomeCientifico"));
                            u.setLocalizacao(obj.getString("Localizacao"));
                            u.setAmbientePropicio(obj.getString("AmbientePropicio"));
                            u.setCicloVida(obj.getString("CicloVida"));
                            u.setInjurias(obj.getString("Injurias"));
                            u.setObservacoes(obj.getString("Observacoes"));
                            u.setHorarioDeAtuacao(obj.getString("HorarioDeAtuacao"));
                            u.setEstagioDeAtuacao(obj.getString("EstagioDeAtuacao"));
                            u.setControleCultural(obj.getString("ControleCultural"));
                            a.setCod_Atinge(obj.getInt("Cod_Atinge"));
                            a.setNivelDeControle(obj.getDouble("NivelDeControle"));
                            a.setNumeroPlantasAmostradas(obj.getInt("NumeroPlantasAmostradas"));
                            a.setPontosPorTalhao(obj.getInt("PontosPorTalhao"));
                            a.setPlantasPorPonto(obj.getInt("PlantasPorPonto"));
                            a.setNumAmostras(obj.getInt("NumAmostras"));
                            a.setFk_Cod_Planta(obj.getInt("fk_Planta_Cod_Planta"));
                            a.setFk_Cod_Praga(obj.getInt("fk_Praga_Cod_Praga"));

                            webPraga.add(u);
                            cp.addPraga(u);
                            ca.addAtinge(a);
                        }
                        localPraga = cp.getPragaAtinge(Cod_Planta);

                        if(webPraga.size() == localPraga.size()) {
                            if(!webPraga.equals(localPraga)) {
                                localPraga = webPraga;
                                for (int i = 0; i < localPraga.size(); i++) {
                                    cp.removerPragaEspecifica(localPraga.get(i));
                                    cp.addPraga(localPraga.get(i));
                                }
                            }else{

                            }
                        }else{
                            localPraga = webPraga;
                            for (int i = 0; i < localPraga.size(); i++) {
                                cp.removerPragaEspecifica(localPraga.get(i));
                                cp.addPraga(localPraga.get(i));
                            }
                        }
                        mDialog.dismiss();
                    } catch (JSONException e) {
                        Toast.makeText(Talhoes.this, e.toString(), Toast.LENGTH_LONG).show();
                        mDialog.dismiss();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(Talhoes.this,error.toString(), Toast.LENGTH_LONG).show();
                    mDialog.dismiss();
                }
            }));

        }
    }

    private void iniciarRecyclerView(){

        Log.d(TAG, "iniciarRecyclerView:  init iniciar");
        RecyclerView rv = findViewById(R.id.RVTalhoes);
        TalhaoCardAdapter adapter = new TalhaoCardAdapter(this, cards, codCultura,Cod_Propriedade, nomePropriedade, nome,Cod_Planta);

        rv.setAdapter(adapter);
        rv.setLayoutManager(new LinearLayoutManager(this));

    }

    public void openDialog(){
        mDialog = new Dialog(this);
        //vamos remover o titulo da Dialog
        mDialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        //vamos carregar o xml personalizado
        mDialog.setContentView(R.layout.dialog);
        //DEixamos transparente
        mDialog.getWindow().setBackgroundDrawable(new ColorDrawable(android.graphics.Color.TRANSPARENT));
        // não permitimos fechar esta dialog
        mDialog.setCancelable(false);
        //temos a instancia do ProgressBar!
        final ProgressBar progressBar = ProgressBar.class.cast(mDialog.findViewById(R.id.progressBar));

        mDialog.show();

        // mDialog.dismiss(); -> para fechar a dialog

    }
}
